import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { NavigateLink } from 'components';
import { UnauthorizedLayout } from 'layouts/UnauthorizedLayout';

import { useAppDispatch } from 'hooks';
import { ROUTES } from 'routes';
import { useEntraCallbackMutation } from 'services/auth';

import { AuthErrorMessage } from 'App/AuthErrorMessage';
import { getBaseUrl } from 'App/helpers';
import { Loading } from 'App/Loading';
import { setAuthData } from 'App/slice';

export const LoginByEntraIDCallback: React.FC = () => {
    const { t } = useTranslation();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const code = searchParams.get('code');
    const state = searchParams.get('state');
    const [isInvalidCode, setIsInvalidCode] = useState(false);
    const dispatch = useAppDispatch();

    const [entraCallback] = useEntraCallbackMutation();

    const checkCode = () => {
        if (code && state) {
            entraCallback({ code, state, base_url: getBaseUrl() })
                .unwrap()
                .then(({ creds: { token } }) => {
                    dispatch(setAuthData({ token }));
                    navigate('/');
                })
                .catch(() => {
                    setIsInvalidCode(true);
                });
        }
    };

    useEffect(() => {
        if (code && state) {
            checkCode();
        } else {
            setIsInvalidCode(true);
        }
    }, []);

    if (isInvalidCode)
        return (
            <UnauthorizedLayout>
                <AuthErrorMessage title={t('auth.authorization_failed')}>
                    <NavigateLink href={ROUTES.BASE}>{t('auth.try_again')}</NavigateLink>
                </AuthErrorMessage>
            </UnauthorizedLayout>
        );

    return (
        <UnauthorizedLayout>
            <Loading />;
        </UnauthorizedLayout>
    );
};
